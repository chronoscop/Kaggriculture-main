//! Owned tensors, with errors and RAII at the Rust boundary. No Python interpreter.
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    marker::PhantomData,
    rc::Rc,
};
type Raw = *mut c_void;
extern "C" {
    fn ft_error() -> *const c_char;
    fn ft_threads(n: c_int);
    fn ft_worker_threads(n: c_int);
    fn ft_thread_count() -> c_int;
    fn ft_grad_mode(enabled: c_int) -> c_int;
    fn ft_free(t: Raw);
    fn ft_numel(t: Raw) -> i64;
    fn ft_new(
        values: *const c_void,
        dims: *const i64,
        ndims: c_int,
        kind: c_int,
        device: c_int,
        grad: c_int,
    ) -> Raw;
    fn ft_data(t: Raw, output: *mut f32, count: i64) -> c_int;
    fn ft_copy(dst: Raw, src: Raw) -> c_int;
    fn ft_backward(loss: Raw) -> c_int;
    fn ft_zero_grad(t: Raw);
    fn ft_has_grad(t: Raw) -> c_int;
    fn ft_op(
        op: c_int,
        inputs: *const Raw,
        count: c_int,
        ints: *const i64,
        ni: c_int,
        scalars: *const f64,
        ns: c_int,
    ) -> Raw;
}
fn error() -> String {
    unsafe { CStr::from_ptr(ft_error()) }
        .to_string_lossy()
        .into_owned()
}
pub struct Tensor {
    raw: Raw,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { ft_free(self.raw) }
    }
}
pub struct NoGrad(c_int);
impl NoGrad {
    pub fn new() -> Self {
        Self(unsafe { ft_grad_mode(0) })
    }
}
impl Default for NoGrad {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for NoGrad {
    fn drop(&mut self) {
        unsafe {
            ft_grad_mode(self.0);
        }
    }
}
pub fn threads(n: i32) {
    // LibTorch inter-op configuration is process-global and may only be set once.
    static INIT: std::sync::Once = std::sync::Once::new();
    assert!(n > 0);
    INIT.call_once(|| unsafe { ft_threads(n) });
}
/// OpenMP/MKL thread settings must also be initialized on each Rust worker.
/// Inter-op remains process-global and is initialized once by threads().
pub fn worker_threads() {
    unsafe {
        ft_worker_threads(1);
    }
}
pub fn thread_count() -> i32 {
    unsafe { ft_thread_count() }
}
impl Tensor {
    fn own(raw: Raw) -> Result<Self, String> {
        if raw.is_null() {
            Err(error())
        } else {
            Ok(Self {
                raw,
                _thread: PhantomData,
            })
        }
    }
    fn shape_count(shape: &[i64]) -> Result<usize, String> {
        shape.iter().try_fold(1usize, |n, &d| {
            if d < 0 {
                Err("negative tensor dimension".into())
            } else {
                n.checked_mul(d as usize)
                    .ok_or_else(|| "tensor size overflow".into())
            }
        })
    }
    pub fn floats(values: &[f32], shape: &[i64], device: i32, grad: bool) -> Result<Self, String> {
        if Self::shape_count(shape)? != values.len() {
            return Err("float tensor shape mismatch".into());
        }
        Self::own(unsafe {
            ft_new(
                values.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                0,
                device,
                i32::from(grad),
            )
        })
    }
    pub fn integers(
        values: &[i64],
        shape: &[i64],
        device: i32,
        boolean: bool,
    ) -> Result<Self, String> {
        if Self::shape_count(shape)? != values.len() {
            return Err("index tensor shape mismatch".into());
        }
        Self::own(unsafe {
            ft_new(
                values.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                if boolean { 2 } else { 1 },
                device,
                0,
            )
        })
    }
    pub fn operation(
        code: i32,
        inputs: &[&Tensor],
        ints: &[i64],
        scalars: &[f64],
    ) -> Result<Self, String> {
        let (arity, ni, ns) = match code {
            44 | 45 => (4, 0, 0),
            0 | 11 => (3, 0, 0),
            2 => (inputs.len(), 1, 0),
            3 | 4 | 7 | 9 | 23 | 42 => (1, 1, 0),
            5 | 17 | 35 | 36 => (1, ints.len(), 0),
            10 => (2, 0, 1),
            12..=15 | 20 | 46 => (2, 0, 0),
            19 => (1, 0, 2),
            21 | 22 => (2, 1, 0),
            27 => (1, 3, 0),
            29..=31 | 43 => (1, 0, 1),
            39 | 41 => (3, 0, 1),
            40 => (2, 0, 1),
            1 | 6 | 8 | 16 | 18 | 24..=26 | 28 | 32..=34 | 37 | 38 => (1, 0, 0),
            _ => return Err("unknown tensor operation".into()),
        };
        if inputs.len() != arity || ints.len() != ni || scalars.len() != ns {
            return Err("invalid tensor operation arguments".into());
        }
        if inputs.is_empty() {
            return Err("tensor operation needs an input".into());
        }
        let pointers: Vec<_> = inputs.iter().map(|t| t.raw).collect();
        Self::own(unsafe {
            ft_op(
                code,
                pointers.as_ptr(),
                pointers.len() as i32,
                ints.as_ptr(),
                ints.len() as i32,
                scalars.as_ptr(),
                scalars.len() as i32,
            )
        })
    }
    pub fn unary(&self, code: i32) -> Result<Self, String> {
        Self::operation(code, &[self], &[], &[])
    }
    pub fn binary(&self, code: i32, b: &Self) -> Result<Self, String> {
        Self::operation(code, &[self, b], &[], &[])
    }
    pub fn scalar(&self, code: i32, v: f64) -> Result<Self, String> {
        Self::operation(code, &[self], &[], &[v])
    }
    pub fn dim(&self, code: i32, dim: i64) -> Result<Self, String> {
        Self::operation(code, &[self], &[dim], &[])
    }
    pub fn data(&self) -> Result<Vec<f32>, String> {
        let n = unsafe { ft_numel(self.raw) };
        let mut out = vec![0.; n as usize];
        if unsafe { ft_data(self.raw, out.as_mut_ptr(), n) } != 0 {
            return Err(error());
        }
        Ok(out)
    }
    pub fn value(&self) -> Result<f64, String> {
        let x = self.data()?;
        if x.len() != 1 {
            return Err("expected scalar tensor".into());
        }
        Ok(x[0] as f64)
    }
    pub fn copy_from(&mut self, source: &Self) -> Result<(), String> {
        if unsafe { ft_copy(self.raw, source.raw) } != 0 {
            Err(error())
        } else {
            Ok(())
        }
    }
    pub fn backward(&self) -> Result<(), String> {
        if unsafe { ft_backward(self.raw) } != 0 {
            Err(error())
        } else {
            Ok(())
        }
    }
    pub fn has_grad(&self) -> bool {
        unsafe { ft_has_grad(self.raw) != 0 }
    }
    pub fn zero_grad(&mut self) {
        unsafe { ft_zero_grad(self.raw) }
    }
}

#[cfg(test)]
mod worker_tests {
    #[test]
    fn single_category_distribution_matches_hierarchical_distribution() {
        super::threads(1);
        let devices = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
            vec![-1, 0]
        } else {
            vec![-1]
        };
        for device in devices {
            let scores =
                super::Tensor::floats(&[0.2, 1.3, -0.7, 2., 0., 3.], &[2, 3], device, false)
                    .unwrap();
            let mask = super::Tensor::integers(&[1, 1, 1, 1, 1, 0], &[2, 3], device, true).unwrap();
            let groups =
                super::Tensor::integers(&[1, 1, 1, 1, 1, 0], &[2, 3], device, false).unwrap();
            let head = super::Tensor::floats(&[0.3; 38], &[2, 19], device, false).unwrap();
            let full = super::Tensor::operation(44, &[&scores, &head, &groups, &mask], &[], &[])
                .unwrap()
                .data()
                .unwrap();
            let flat = super::Tensor::operation(46, &[&scores, &mask], &[], &[])
                .unwrap()
                .data()
                .unwrap();
            for (a, b) in full.iter().zip(flat) {
                assert!((*a - b).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn rust_worker_uses_one_tensor_thread() {
        super::threads(1);
        assert_eq!(
            std::thread::spawn(|| {
                super::worker_threads();
                super::thread_count()
            })
            .join()
            .unwrap(),
            1
        );
    }
}
