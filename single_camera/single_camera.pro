#-------------------------------------------------
#
# libCalib example single camera calibration
#
#-------------------------------------------------

QT       -= core gui

TARGET = single_camera
TEMPLATE = app

CONFIG += c++14

SOURCES += \
        main.cpp

win32 {
    CONFIG += static_runtime
}


include(../dependencies.pri)
