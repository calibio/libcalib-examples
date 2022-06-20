#-------------------------------------------------
#
# libCalib example step target calibration
#
#-------------------------------------------------

QT       -= core gui

TARGET = step_target
TEMPLATE = app

CONFIG += c++14

SOURCES += \
        main.cpp

win32 {
    CONFIG += static_runtime
}


include(../dependencies.pri)
