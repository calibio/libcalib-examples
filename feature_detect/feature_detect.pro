#-------------------------------------------------
#
# libCalib example feature detection
#
#-------------------------------------------------

QT       -= core gui

TARGET = feature_detect
TEMPLATE = app

CONFIG += c++14

SOURCES += \
        main.cpp


win32 {
    CONFIG += static_runtime
}


include(../dependencies.pri)
