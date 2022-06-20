#-------------------------------------------------
#
# libCalib example OneShot target calibration
#
#-------------------------------------------------

QT       -= core gui

TARGET = one_shot
TEMPLATE = app

CONFIG += c++14

SOURCES += \
        main.cpp

win32 {
    CONFIG += static_runtime
}


include(../dependencies.pri)
