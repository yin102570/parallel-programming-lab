@echo off
REM ============================================================
REM  Windows端: 打包所有源文件为可上传的tar包
REM  上传到AUP后解压即可运行
REM ============================================================

cd /d "%~dp0"

REM 创建临时目录
mkdir __pack_temp 2>nul
copy flat_scan_gpu_hip.h __pack_temp\ >nul
copy flat_scan_gpu_hip.cpp __pack_temp\ >nul
copy main_ann_gpu.cc __pack_temp\ >nul
copy Makefile __pack_temp\ >nul
copy setup_and_run.sh __pack_temp\ >nul

REM 打包
tar -cvf gpu_ann_package.tar -C __pack_temp .
del /q __pack_temp\* 2>nul
rmdir __pack_temp 2>nul

echo.
echo ============================================
echo  已生成: gpu_ann_package.tar
echo  将此文件 + 数据集一起上传到AUP服务器:
echo  
echo    1. 上传 gpu_ann_package.tar 到 ~/gpu_ann/
echo    2. 上传 DEEP100K* 文件到 ~/gpu_ann/data/
echo    3. 执行:
echo       cd ~/gpu_ann
echo       tar xvf gpu_ann_package.tar
echo       chmod +x setup_and_run.sh
echo       bash setup_and_run.sh ./data
echo ============================================
pause
