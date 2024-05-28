use anyhow::Result;
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

use crate::ImageBuffer;

#[derive(Clone, Copy)]
enum AccessMethods {
    Read,
    Write,
    Search,
}

pub struct SharedBufferLock<'a, T> {
    _id: usize,
    lock: MutexGuard<'a, T>,
    method: AccessMethods,
    shared_buffer: &'a SharedBuffer<T>,
}

impl<T> Deref for SharedBufferLock<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &*self.lock
    }
}

impl<T> DerefMut for SharedBufferLock<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.lock
    }
}

impl<T> Drop for SharedBufferLock<'_, T> {
    fn drop(&mut self) {
        match self.method {
            AccessMethods::Read => self.shared_buffer.read_finish(self._id),
            AccessMethods::Write => self.shared_buffer.write_finish(self._id),
            AccessMethods::Search => self.shared_buffer.search_finish(self._id),
        }
    }
}

struct BufferInfo {
    lfu: usize,
    occupied: bool,
    is_new: bool,
}

impl Default for BufferInfo {
    fn default() -> Self {
        BufferInfo {
            lfu: 0,
            occupied: false,
            is_new: false,
        }
    }
}

impl Clone for BufferInfo {
    fn clone(&self) -> Self {
        BufferInfo {
            lfu: self.lfu,
            occupied: self.occupied,
            is_new: self.is_new,
        }
    }
}

impl Copy for BufferInfo {}

pub struct SharedBuffer<T> {
    message_cond: Arc<Condvar>,
    info: Mutex<Vec<BufferInfo>>,
    buffers: Vec<Mutex<T>>,
}

impl<T> SharedBuffer<T> {
    pub fn new(reader_writer_cnt: usize, f: impl Fn() -> Result<T>) -> Result<Self> {
        let mut vec = Vec::with_capacity(reader_writer_cnt);
        for _ in 0..reader_writer_cnt {
            vec.push(Mutex::new(f()?));
        }
        Ok(SharedBuffer {
            message_cond: Arc::new(Condvar::new()),
            info: Mutex::new(vec![BufferInfo::default(); reader_writer_cnt]),
            buffers: vec,
        })
    }

    #[inline]
    fn get_buffer_info(&self) -> MutexGuard<Vec<BufferInfo>> {
        match self.info.lock() {
            Ok(info) => info,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[inline]
    fn check_condition(&self, info: &MutexGuard<Vec<BufferInfo>>) -> bool {
        info.iter()
            .fold(0, |acc, x| acc + if x.is_new { 1 } else { 0 })
            != 0
    }

    #[inline]
    fn get_read_index(&self) -> Option<usize> {
        let mut info = self.get_buffer_info();
        if !self.check_condition(&info) {
            info = self
                .message_cond
                .wait(info)
                .unwrap_or_else(|x| x.into_inner());
            if !self.check_condition(&info) {
                return None;
            }
        }

        let index = info
            .iter()
            .enumerate()
            .filter(|x| !x.1.occupied)
            .min_by_key(|x| x.1.lfu)
            .unwrap()
            .0;
        info[index].is_new = false;
        info[index].occupied = true;
        Some(index)
    }

    #[inline]
    fn get_write_index(&self) -> usize {
        let mut info = self.get_buffer_info();
        let index = info
            .iter()
            .enumerate()
            .filter(|x| !x.1.occupied)
            .max_by_key(|x| x.1.lfu)
            .unwrap()
            .0;
        info[index].occupied = true;
        index
    }

    #[inline]
    fn get_buffer(&self, index: usize) -> MutexGuard<T> {
        match self.buffers[index].lock() {
            Ok(buffer) => buffer,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub fn read(&self) -> Option<SharedBufferLock<T>> {
        let index = self.get_read_index()?;
        Some(SharedBufferLock {
            _id: index,
            method: AccessMethods::Read,
            lock: self.get_buffer(index),
            shared_buffer: self,
        })
    }

    fn read_finish(&self, id: usize) {
        let mut info = self.get_buffer_info();
        info[id].occupied = false;
    }

    pub fn write(&self) -> SharedBufferLock<T> {
        let index = self.get_write_index();
        SharedBufferLock {
            _id: index,
            method: AccessMethods::Write,
            lock: self.get_buffer(index),
            shared_buffer: self,
        }
    }

    fn write_finish(&self, id: usize) {
        let mut info = self.get_buffer_info();
        info.iter_mut().for_each(|x| x.lfu += 1);
        info[id].lfu = 0;
        info[id].occupied = false;
        info[id].is_new = true;
        drop(info);

        self.message_cond.notify_one();
    }

    fn search_finish(&self, id: usize) {
        let mut info = self.get_buffer_info();
        info[id].occupied = false;
    }
}

impl SharedBuffer<ImageBuffer> {
    pub fn search(&self, timestamp: std::time::Instant) -> Option<SharedBufferLock<ImageBuffer>> {
        let (idx, lock) = self.buffers.iter().enumerate().find(|x| {
            if let Ok(lock) = x.1.try_lock() {
                lock.timestamp == timestamp
            } else {
                false
            }
        })?;

        let mut info = self.get_buffer_info();
        info[idx].occupied = true;
        drop(info);
        Some(SharedBufferLock {
            _id: idx,
            method: AccessMethods::Search,
            lock: lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            shared_buffer: self,
        })
    }
}

pub struct Reader<T>(Arc<SharedBuffer<T>>);
impl<T> Reader<T> {
    pub fn read(&self) -> Option<SharedBufferLock<'_, T>> {
        self.0.read()
    }
}

pub struct Searcher<T>(Arc<SharedBuffer<T>>);
impl Searcher<ImageBuffer> {
    pub fn search(&self, timestamp: std::time::Instant) -> Option<SharedBufferLock<ImageBuffer>> {
        self.0.search(timestamp)
    }
}

pub struct Writer<T>(Arc<SharedBuffer<T>>);
impl<T> Writer<T> {
    pub fn new(reader_writer_cnt: usize, f: impl Fn() -> Result<T>) -> Result<Self> {
        Ok(Self(Arc::new(SharedBuffer::new(reader_writer_cnt, f)?)))
    }

    pub fn get_reader(&self) -> Reader<T> {
        Reader(self.0.clone())
    }

    pub fn get_searcher(&self) -> Searcher<T> {
        Searcher(self.0.clone())
    }

    pub fn write(&self) -> SharedBufferLock<'_, T> {
        self.0.write()
    }
}

impl<T> Drop for Writer<T> {
    fn drop(&mut self) {
        self.0.message_cond.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    #[test]
    fn test_shared_buffer_spsc() {
        let n = 1000000;
        let buffer = SharedBuffer::new(10, || Ok(0usize)).unwrap();
        let share_buffer1 = Arc::new(buffer);
        let share_buffer2 = share_buffer1.clone();
        let buffer = share_buffer2.clone();
        let handle1 = thread::spawn(move || {
            for _ in 0..n - 1 {
                #[allow(unused)]
                let read_buffer = share_buffer1.read();
            }
        });
        let handle2 = thread::spawn(move || {
            for i in 0..n {
                let mut write_buffer = share_buffer2.write();
                *write_buffer = i;
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
        assert_eq!(n - 1, *buffer.read().unwrap());
    }

    #[test]
    fn test_shared_buffer_mpsc() {
        let n = 1000000;
        let buffer = SharedBuffer::new(10, || Ok(0usize)).unwrap();
        let share_buffer1 = Arc::new(buffer);
        let share_buffer2 = share_buffer1.clone();
        let share_buffer3 = share_buffer1.clone();
        let buffer = share_buffer1.clone();
        let handle1 = thread::spawn(move || {
            for _ in 0..n {
                #[allow(unused)]
                let read_buffer = share_buffer1.read();
            }
        });
        let handle2 = thread::spawn(move || {
            for i in 0..n {
                let mut write_buffer = share_buffer2.write();
                *write_buffer = i;
            }
        });
        let handle3 = thread::spawn(move || {
            for i in 1..n + 1 {
                let mut write_buffer = share_buffer3.write();
                *write_buffer = i;
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();
        let latest = buffer.read().unwrap();
        assert!(n - 1 == *latest || n == *latest);
    }
}
