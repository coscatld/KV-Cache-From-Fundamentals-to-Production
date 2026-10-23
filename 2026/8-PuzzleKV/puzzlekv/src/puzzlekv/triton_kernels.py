"""Two-stage kernels. FP32 accumulation; optional import for non-CUDA hosts."""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _mm(X,Y,Z,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr=16,BN:tl.constexpr=32,BK:tl.constexpr=32):
        rows=tl.program_id(0)*BM+tl.arange(0,BM)
        cols=tl.program_id(1)*BN+tl.arange(0,BN)
        ks=tl.arange(0,BK)
        acc=tl.full((BM,BN),0,tl.float32)
        for block in range(tl.cdiv(K,BK)):
            kk=block*BK+ks
            x=tl.load(X+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),0)
            y=tl.load(Y+kk[:,None]*N+cols[None,:],(kk[:,None]<K)&(cols[None,:]<N),0)
            acc+=tl.dot(x.to(tl.float32),y.to(tl.float32),input_precision="ieee")
        tl.store(Z+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))

    @triton.jit
    def _transpose(X,Y,M:tl.constexpr,N:tl.constexpr,B:tl.constexpr=256):
        i=tl.program_id(0)*B+tl.arange(0,B)
        value=tl.load(X+i,i<M*N,0)
        tl.store(Y+(i%N)*M+i//N,value,i<M*N)

def _check(*xs):
    if triton is None:
        raise RuntimeError("Install Triton in a CUDA environment")
    if not all(x.is_cuda and x.ndim==2 and x.is_contiguous() for x in xs):
        raise ValueError("expected contiguous CUDA matrices")
    if len({x.device for x in xs}) != 1:
        raise ValueError("device mismatch")

def _transpose_matrix(x):
    y=torch.empty((x.shape[1],x.shape[0]),device=x.device,dtype=x.dtype)
    _transpose[(triton.cdiv(x.numel(),256),)](x,y,*x.shape)
    return y

def _matmul(x,y):
    if x.shape[1]!=y.shape[0]:
        raise ValueError("matrix shape mismatch")
    m,k=x.shape; n=y.shape[1]
    z=torch.empty((m,n),device=x.device,dtype=torch.float32)
    _mm[(triton.cdiv(m,16),triton.cdiv(n,32))](x,y,z,m,n,k,num_warps=4)
    return z

def lowrank_logits(q,a,b):
    _check(q,a,b)
    return _matmul(_matmul(q,_transpose_matrix(b)),_transpose_matrix(a))

def lowrank_value(weights,a,b):
    _check(weights,a,b)
    return _matmul(_matmul(weights,a),b)
