#!/bin/bash

gcc -D__CUDA_ARCH__=300 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  -I"../common/inc/" "-I/usr/local/cuda-9.0/bin/..//include"   -D"__CUDACC_VER_BUILD__=176" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=9" -include "cuda_runtime.h"-m64 "vectorAdd.cu" > "/tmp/tmpxft_000083df_00000000-8_vectorAdd.cpp1.ii"

cicc --gnu_version=50400 --allow_managed   -arch compute_30 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_000083df_00000000-2_vectorAdd.fatbin.c" -tused -nvvmir-library "/usr/local/cuda-9.0/bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_000083df_00000000-3_vectorAdd.module_id" --orig_src_file_name "vectorAdd.cu" --gen_c_file_name "/tmp/tmpxft_000083df_00000000-5_vectorAdd.cudafe1.c" --stub_file_name "/tmp/tmpxft_000083df_00000000-5_vectorAdd.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_000083df_00000000-5_vectorAdd.cudafe1.gpu"  "/tmp/tmpxft_000083df_00000000-8_vectorAdd.cpp1.ii" -o "/tmp/tmpxft_000083df_00000000-5_vectorAdd.ptx"  #ignore the ptx file created here

ptxas -arch=sm_30 -m64  "vectorAdd.ptx"  -o "/tmp/tmpxft_000083df_00000000-9_vectorAdd.sm_30.cubin"

# use our modified ptx file
fatbinary --create="/tmp/tmpxft_000083df_00000000-2_vectorAdd.fatbin" -64 "--image=profile=sm_30,file=/tmp/tmpxft_000083df_00000000-9_vectorAdd.sm_30.cubin" "--image=profile=compute_30,file=vectorAdd.ptx" --embedded-fatbin="/tmp/tmpxft_000083df_00000000-2_vectorAdd.fatbin.c" --cuda

rm /tmp/tmpxft_000083df_00000000-2_vectorAdd.fatbin

gcc -E -x c++ -D__CUDACC__ -D__NVCC__  -I"../common/inc/" "-I/usr/local/cuda-9.0/bin/..//include"   -D"__CUDACC_VER_BUILD__=176" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=9" -include "cuda_runtime.h" -m64 "vectorAdd.cu" > "/tmp/tmpxft_000083df_00000000-4_vectorAdd.cpp4.ii"

cudafe++ --gnu_version=50400 --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_000083df_00000000-5_vectorAdd.cudafe1.cpp" --stub_file_name "tmpxft_000083df_00000000-5_vectorAdd.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_000083df_00000000-3_vectorAdd.module_id" "/tmp/tmpxft_000083df_00000000-4_vectorAdd.cpp4.ii"

gcc -D__CUDA_ARCH__=300 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -I"../common/inc/" "-I/usr/local/cuda-9.0/bin/..//include"   -m64 -o "/tmp/tmpxft_000083df_00000000-10_vectorAdd.o" "/tmp/tmpxft_000083df_00000000-5_vectorAdd.cudafe1.cpp"

nvlink --arch=sm_30 --register-link-binaries="/tmp/tmpxft_000083df_00000000-6_a_dlink.reg.c" -m64   "-L/usr/local/cuda-9.0/bin/..//lib64/stubs" "-L/usr/local/cuda-9.0/bin/..//lib64" -cpu-arch=X86_64 "/tmp/tmpxft_000083df_00000000-10_vectorAdd.o"  -o "/tmp/tmpxft_000083df_00000000-11_a_dlink.sm_30.cubin"

fatbinary --create="/tmp/tmpxft_000083df_00000000-7_a_dlink.fatbin" -64 -link "--image=profile=sm_30,file=/tmp/tmpxft_000083df_00000000-11_a_dlink.sm_30.cubin" --embedded-fatbin="/tmp/tmpxft_000083df_00000000-7_a_dlink.fatbin.c"

rm /tmp/tmpxft_000083df_00000000-7_a_dlink.fatbin

gcc -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_000083df_00000000-7_a_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_000083df_00000000-6_a_dlink.reg.c\"" -I. -I"../common/inc/" "-I/usr/local/cuda-9.0/bin/..//include"   -D"__CUDACC_VER_BUILD__=176" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=9" -m64 -o "/tmp/tmpxft_000083df_00000000-12_a_dlink.o" "/usr/local/cuda-9.0/bin/crt/link.stub"

g++ -m64 -o "a.out" -Wl,--start-group "/tmp/tmpxft_000083df_00000000-12_a_dlink.o" "/tmp/tmpxft_000083df_00000000-10_vectorAdd.o"   "-L/usr/local/cuda-9.0/bin/..//lib64/stubs" "-L/usr/local/cuda-9.0/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread -ldl  -Wl,--end-group
