use std::os::raw::c_char;

pub unsafe fn write_err(msg: &str, buf: *mut c_char, len: usize) {
    if !buf.is_null() && len > 0 {
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(len - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buf, copy_len);
            *buf.add(copy_len) = 0;
        }
    }
}
