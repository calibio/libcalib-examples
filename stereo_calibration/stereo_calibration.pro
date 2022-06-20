#-------------------------------------------------
#
# libCalib example stereo camera calibration
#
#-------------------------------------------------

QT       -= core gui

TARGET = stereo_calibration
TEMPLATE = app

CONFIG += c++14

SOURCES += \
        main.cpp

win32 {
    CONFIG += static_runtime
}


include(../dependencies.pri)
