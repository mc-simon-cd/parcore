// Copyright 2026 mcsimon
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Memory Module
//!
//! Provides a **Unified Memory** abstraction. `SharedBuffer<T>` wraps a
//! heap-allocated slice behind `Arc<RwLock<…>>`, giving any number of threads
//! concurrent read access and exclusive write access — no explicit `memcpy`
//! between "device" and "host" required.

pub mod buffer;
pub use buffer::SharedBuffer;
