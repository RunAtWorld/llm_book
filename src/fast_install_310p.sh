#reploace yum sources
# cp /etc/yum.repos.d/openEuler.repo /etc/yum.repos.d/openEuler.repo.bak
# cat > /etc/yum.repos.d/openEuler.repo <<-EOF
# #huaweicloud
# [openEuler-everything]
# name=openEuler-everything
# baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-24.09/everything/x86_64/
# enabled=1
# gpgcheck=0
# gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-24.09/everything/x86_64/RPM-GPG-KEY-openEuler
        
# [openEuler-EPOL]
# name=openEuler-epol
# baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-24.09/EPOL/main/x86_64/
# enabled=1
# gpgcheck=0

# [openEuler-update]
# name=openEuler-update
# baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-24.09/update/x86_64/
# enabled=1
# gpgcheck=0
# EOF
# yum clean all
# yum makecache

#install driver
mkdir ~/driver
cd ~/pkg/driver
#download dirver
curl https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC1/Ascend-hdk-310p-npu-firmware_7.1.0.6.220.run?response-content-type=application/octet-stream -o Ascend-hdk-310p-npu-firmware_7.1.0.6.220.run
curl https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC1/Ascend-hdk-310p-npu-driver_24.1.rc1_linux-aarch64.run?response-content-type=application/octet-stream -o Ascend-hdk-310p-npu-driver_24.1.rc1_linux-aarch64.run
yum install -y make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r)
chmod a+x *.run
./Ascend-hdk-*-npu-driver_*_linux-aarch64.run --full --install-for-all --quiet
./Ascend-hdk-*-npu-firmware_*.run --full --install-for-all --quiet

#install cann
mkdir ~/pkg
cd ~/pkg/cann
#download cann
curl https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run?response-content-type=application/octet-stream -o Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run
curl https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-kernels-310p_8.0.RC3_linux-aarch64.run?response-content-type=application/octet-stream -o Ascend-cann-kernels-310p_8.0.RC3_linux-aarch64.run
chmod a+x *.run
./Ascend-cann-toolkit_*_linux-*.run --install --quiet --nox11
./Ascend-cann-kernels-*_linux-*.run --install --quiet --nox11

# mount disk
mkdir /data
mkfs.xfs /dev/vdb
mount /dev/vdb /data

# install docker 
yum install docker
cat > /etc/docker/daemon.json <<-EOF
{
    "data-root": "/data/docker",
    "registry-mirrors" : [
        "https://05092a8eb30025720fa9c01e8cefab80.mirror.swr.myhuaweicloud.com",
        "https://docker.registry.cyou",
        "https://docker-cf.registry.cyou",
        "https://dockercf.jsdelivr.fyi",
        "https://docker.jsdelivr.fyi",
        "https://dockertest.jsdelivr.fyi",
        "https://mirror.aliyuncs.com",
        "https://dockerproxy.com",
        "https://mirror.baidubce.com",
        "https://docker.m.daocloud.io",
        "https://docker.nju.edu.cn",
        "https://docker.mirrors.sjtug.sjtu.edu.cn",
        "https://docker.mirrors.ustc.edu.cn",
        "https://mirror.iscas.ac.cn",
        "https://docker.rainbond.cc",
        "https://do.nark.eu.org",
        "https://dc.j8.work",
        "https://dockerproxy.com",
        "https://gst6rzl9.mirror.aliyuncs.com",
        "https://registry.docker-cn.com",
        "http://hub-mirror.c.163.com",
        "http://mirrors.ustc.edu.cn/",
        "https://mirrors.tuna.tsinghua.edu.cn/",
        "http://mirrors.sohu.com/"
    ],
    "insecure-registries" : [
        "registry.docker-cn.com",
        "docker.mirrors.ustc.edu.cn"
    ]
}
EOF
systemctl reload docker
systemctl restart docker
docker info