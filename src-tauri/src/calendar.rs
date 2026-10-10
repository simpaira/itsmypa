//! The home screen's "your meeting starts now — record it?" prompt. The
//! calendar is fetched at most every 5 minutes (the screen asks every minute).

use itsmypa_engine::calendar::{self, Event};
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const REFRESH: Duration = Duration::from_secs(300);

#[derive(Default)]
pub struct Calendar {
    cache: Mutex<Option<Cached>>,
}

struct Cached {
    url: String,
    at: Instant,
    events: Vec<Event>,
    error: Option<String>,
}

impl Calendar {
    pub fn upcoming(&self, url: &str) -> Value {
        if url.is_empty() {
            return json!({"configured": false, "events": [], "error": null});
        }
        let mut cache = self.cache.lock().unwrap();
        let fresh = cache.as_ref().is_some_and(|c| c.url == url && c.at.elapsed() < REFRESH);
        if !fresh {
            let (events, error) = match calendar::upcoming(url) {
                Ok(ev) => (ev, None),
                Err(e) => (Vec::new(), Some(format!("{e:#}"))),
            };
            *cache = Some(Cached { url: url.to_string(), at: Instant::now(), events, error });
        }
        let c = cache.as_ref().unwrap();
        json!({"configured": true, "events": c.events, "error": c.error})
    }
}
