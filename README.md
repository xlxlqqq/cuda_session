# cuda并行编程学习

### session_1 
hello world demo

nvcc \     
  -ccbin=g++-10 \
  -std=c++14 \
  matrixAdd.cu \
  -o matrixAdd
