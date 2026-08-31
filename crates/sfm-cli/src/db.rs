//! The project's `database.sqlite`: one row per image (with its detected
//! `FeatureSet`, bincode-packed into a BLOB) plus the camera each image
//! belongs to. `sfm match` reads this back to run pairing/matching; `sfm map`
//! reads matches to build the reconstruction. Using sqlite (not our own
//! binary format) means the intermediate state is inspectable with any
//! off-the-shelf sqlite browser, which matters for a no-GUI tool.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use sfm_core::{Camera, FeatureSet, TwoViewGeometryRecord};

pub struct Database {
    pub conn: Connection,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        let db = Database { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                camera_id INTEGER NOT NULL,
                name TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                -- Dataset-layout provenance (see `dataset::DiscoveredImage`).
                -- Kept so later stages and reports can name a feature by where
                -- it came from; `-1` marks a merged row spanning captures.
                capture_id INTEGER NOT NULL DEFAULT 0,
                phys_camera_id INTEGER NOT NULL DEFAULT 0,
                image_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(camera_id) REFERENCES cameras(id)
            );
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY,
                detector TEXT NOT NULL,
                num_features INTEGER NOT NULL,
                data BLOB NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id)
            );
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                image_id1 INTEGER NOT NULL,
                image_id2 INTEGER NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (image_id1, image_id2)
            );
            ",
        )?;
        Ok(())
    }

    pub fn upsert_camera(&self, camera: &Camera) -> Result<()> {
        let params_json = serde_json::to_string(&camera.model.params())?;
        self.conn.execute(
            "INSERT INTO cameras (id, model, width, height, params) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET model=excluded.model, width=excluded.width, height=excluded.height, params=excluded.params",
            params![camera.camera_id, camera.model.name(), camera.width, camera.height, params_json],
        )?;
        Ok(())
    }

    /// Reuse an existing camera row for these exact dimensions if one
    /// exists, so re-running `sfm extract` doesn't spawn a duplicate camera
    /// (and duplicate intrinsics estimate) every time.
    pub fn find_camera_id_by_dims(&self, width: u32, height: u32) -> Result<Option<u32>> {
        let id: Option<u32> = self
            .conn
            .query_row(
                "SELECT id FROM cameras WHERE width = ?1 AND height = ?2 LIMIT 1",
                params![width, height],
                |row| row.get(0),
            )
            .ok();
        Ok(id)
    }

    pub fn find_image_id_by_name(&self, name: &str) -> Result<Option<u32>> {
        let id: Option<u32> = self
            .conn
            .query_row(
                "SELECT id FROM images WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .ok();
        Ok(id)
    }

    pub fn max_camera_id(&self) -> Result<u32> {
        Ok(self
            .conn
            .query_row("SELECT COALESCE(MAX(id), 0) FROM cameras", [], |row| {
                row.get(0)
            })?)
    }

    pub fn max_image_id(&self) -> Result<u32> {
        Ok(self
            .conn
            .query_row("SELECT COALESCE(MAX(id), 0) FROM images", [], |row| {
                row.get(0)
            })?)
    }

    pub fn list_cameras(&self) -> Result<Vec<Camera>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, model, width, height, params FROM cameras")?;
        let rows = stmt.query_map([], |row| {
            let id: u32 = row.get(0)?;
            let model_name: String = row.get(1)?;
            let width: u32 = row.get(2)?;
            let height: u32 = row.get(3)?;
            let params_json: String = row.get(4)?;
            Ok((id, model_name, width, height, params_json))
        })?;
        let mut cameras = Vec::new();
        for row in rows {
            let (id, model_name, width, height, params_json) = row?;
            let params: Vec<f64> = serde_json::from_str(&params_json)?;
            let model = sfm_core::CameraModel::from_name_and_params(&model_name, &params)
                .with_context(|| format!("unknown camera model in database: {model_name}"))?;
            cameras.push(Camera {
                camera_id: id,
                model,
                width,
                height,
            });
        }
        Ok(cameras)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_image(
        &self,
        id: u32,
        camera_id: u32,
        name: &str,
        width: u32,
        height: u32,
        capture_id: i64,
        phys_camera_id: u32,
        image_index: u32,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO images (id, camera_id, name, width, height, capture_id, phys_camera_id, image_index)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET camera_id=excluded.camera_id, name=excluded.name,
               width=excluded.width, height=excluded.height, capture_id=excluded.capture_id,
               phys_camera_id=excluded.phys_camera_id, image_index=excluded.image_index",
            params![id, camera_id, name, width, height, capture_id, phys_camera_id, image_index],
        )?;
        Ok(())
    }

    pub fn list_images(&self) -> Result<Vec<(u32, u32, String, u32, u32)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, camera_id, name, width, height FROM images ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Images with the capture each belongs to, for callers that need to work
    /// per capture rather than over the whole project. `-1` marks a row that
    /// `--merge-multicaps` pooled across captures.
    pub fn list_images_with_capture(&self) -> Result<Vec<(u32, u32, String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, camera_id, name, capture_id FROM images ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn store_features(
        &self,
        image_id: u32,
        detector: &str,
        features: &FeatureSet,
    ) -> Result<()> {
        let data = bincode::serialize(features)?;
        self.conn.execute(
            "INSERT INTO keypoints (image_id, detector, num_features, data) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(image_id) DO UPDATE SET detector=excluded.detector, num_features=excluded.num_features, data=excluded.data",
            params![image_id, detector, features.len() as i64, data],
        )?;
        Ok(())
    }

    pub fn load_features(&self, image_id: u32) -> Result<FeatureSet> {
        let data: Vec<u8> = self.conn.query_row(
            "SELECT data FROM keypoints WHERE image_id = ?1",
            params![image_id],
            |row| row.get(0),
        )?;
        Ok(bincode::deserialize(&data)?)
    }

    pub fn store_geometry(
        &self,
        image_id1: u32,
        image_id2: u32,
        record: &TwoViewGeometryRecord,
    ) -> Result<()> {
        let data = bincode::serialize(record)?;
        let (a, b) = (image_id1.min(image_id2), image_id1.max(image_id2));
        self.conn.execute(
            "INSERT INTO two_view_geometries (image_id1, image_id2, data) VALUES (?1, ?2, ?3)
             ON CONFLICT(image_id1, image_id2) DO UPDATE SET data=excluded.data",
            params![a, b, data],
        )?;
        Ok(())
    }

    pub fn load_geometry(&self, image_id1: u32, image_id2: u32) -> Result<TwoViewGeometryRecord> {
        let (a, b) = (image_id1.min(image_id2), image_id1.max(image_id2));
        let data: Vec<u8> = self.conn.query_row(
            "SELECT data FROM two_view_geometries WHERE image_id1 = ?1 AND image_id2 = ?2",
            params![a, b],
            |row| row.get(0),
        )?;
        Ok(bincode::deserialize(&data)?)
    }

    pub fn list_geometry_pairs(&self) -> Result<Vec<(u32, u32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT image_id1, image_id2 FROM two_view_geometries ORDER BY image_id1, image_id2",
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn clear_geometries(&self) -> Result<()> {
        self.conn.execute("DELETE FROM two_view_geometries", [])?;
        Ok(())
    }
}
