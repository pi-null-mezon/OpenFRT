OpenFRT
===

Set of applications for face recognition

*Sample*

[![fv2db](https://img.youtube.com/vi/diXRtskXeEQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=diXRtskXeEQ)

*Install*

First install 3rdParties:

- [opencv](https://opencv.org/)

- [dlib](http://dlib.net/)

- [qt](https://www.qt.io/)

Then:

```
git clone https://github.com/OpenIRT.git && \
cd OpenIRT && \
docker build -t iface . && \
docker run --name iface -d -p 5000:5000 -v iface:/var/iface iface && \
cd .. && \
git clone https://github.com/OpenFRT.git && \
cd OpenFRT/Apps/fv2db && \
mkdir build && cd build && \
qmake ../fv2db.pro && make -j2 && make install && \
fv2db -ahttp://localhost:5000/iface/identify -s<videostream>
``` 
