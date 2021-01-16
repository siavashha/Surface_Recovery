#-------------------------------------------------
#
# Project created by QtCreator 2015-06-30T19:31:27
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = SurfaceRec_v5
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

#Boost
INCLUDEPATH += /usr/local/include
LIBS += /usr/local/lib/libboost_atomic.so
LIBS += /usr/local/lib/libboost_chrono.so
LIBS += /usr/local/lib/libboost_container.so
LIBS += /usr/local/lib/libboost_context.so
LIBS += /usr/local/lib/libboost_coroutine.so
LIBS += /usr/local/lib/libboost_date_time.so
LIBS += /usr/local/lib/libboost_filesystem.so
LIBS += /usr/local/lib/libboost_graph.so
LIBS += /usr/local/lib/libboost_locale.so
LIBS += /usr/local/lib/libboost_log_setup.so
LIBS += /usr/local/lib/libboost_log.so
LIBS += /usr/local/lib/libboost_math_c99f.so
LIBS += /usr/local/lib/libboost_math_c99l.so
LIBS += /usr/local/lib/libboost_math_c99.so
LIBS += /usr/local/lib/libboost_math_tr1f.so
LIBS += /usr/local/lib/libboost_math_tr1l.so
LIBS += /usr/local/lib/libboost_math_tr1.so
LIBS += /usr/local/lib/libboost_prg_exec_monitor.so
LIBS += /usr/local/lib/libboost_program_options.so
LIBS += /usr/local/lib/libboost_python.so
LIBS += /usr/local/lib/libboost_random.so
LIBS += /usr/local/lib/libboost_regex.so
LIBS += /usr/local/lib/libboost_serialization.so
LIBS += /usr/local/lib/libboost_signals.so
LIBS += /usr/local/lib/libboost_system.so
LIBS += /usr/local/lib/libboost_thread.so
LIBS += /usr/local/lib/libboost_timer.so
LIBS += /usr/local/lib/libboost_unit_test_framework.so
LIBS += /usr/local/lib/libboost_wave.so
LIBS += /usr/local/lib/libboost_wserialization.so

#Eigen
INCLUDEPATH +=/usr/local/include/eigen3

#OpenCV
INCLUDEPATH += /usr/local/include
LIBS += /usr/local/lib/libopencv_core.so
LIBS += /usr/local/lib/libopencv_highgui.so
LIBS += /usr/local/lib/libopencv_imgproc.so
LIBS += /usr/local/lib/libopencv_calib3d.so
LIBS += /usr/local/lib/libopencv_features2d.so
LIBS += /usr/local/lib/libopencv_flann.so
LIBS += /usr/local/lib/libopencv_photo.so
LIBS += /usr/local/lib/libopencv_superres.so
LIBS += /usr/local/lib/libopencv_nonfree.so
LIBS += /usr/local/lib/libopencv_videostab.so
LIBS += /usr/local/lib/libopencv_ml.so
LIBS += /usr/local/lib/libopencv_optim.so
LIBS += /usr/local/lib/libopencv_stitching.so
LIBS += /usr/local/lib/libopencv_objdetect.so
LIBS += /usr/local/lib/libopencv_highgui.so
LIBS += /usr/local/lib/libopencv_video.so
LIBS += /usr/local/lib/libopencv_video.so

#VTK
INCLUDEPATH += /usr/local/include/vtk-6.1
LIBS += /usr/local/lib/libvtkRenderingAnnotation-6.1.so
LIBS += /usr/local/lib/libvtkImagingHybrid-6.1.so
LIBS += /usr/local/lib/libvtkproj4-6.1.so
LIBS += /usr/local/lib/libvtkImagingFourier-6.1.so
LIBS += /usr/local/lib/libvtkCommonSystem-6.1.so
LIBS += /usr/local/lib/libvtklibxml2-6.1.so
LIBS += /usr/local/lib/libvtkexpat-6.1.so
LIBS += /usr/local/lib/libvtktiff-6.1.so
LIBS += /usr/local/lib/libvtkIONetCDF-6.1.so
LIBS += /usr/local/lib/libvtkRenderingVolume-6.1.so
LIBS += /usr/local/lib/libvtkInteractionStyle-6.1.so
LIBS += /usr/local/lib/libvtkViewsCore-6.1.so
LIBS += /usr/local/lib/libvtkCommonTransforms-6.1.so
LIBS += /usr/local/lib/libvtkIOSQL-6.1.so
LIBS += /usr/local/lib/libvtkImagingColor-6.1.so
LIBS += /usr/local/lib/libvtkFiltersAMR-6.1.so
LIBS += /usr/local/lib/libvtkFiltersParallel-6.1.so
LIBS += /usr/local/lib/libvtkhdf5-6.1.so
LIBS += /usr/local/lib/libvtkFiltersModeling-6.1.so
LIBS += /usr/local/lib/libvtkInfovisLayout-6.1.so
LIBS += /usr/local/lib/libvtkViewsInfovis-6.1.so
LIBS += /usr/local/lib/libvtkmetaio-6.1.so
LIBS += /usr/local/lib/libvtksys-6.1.so
LIBS += /usr/local/lib/libvtkNetCDF-6.1.so
LIBS += /usr/local/lib/libvtkRenderingGL2PS-6.1.so
LIBS += /usr/local/lib/libvtkRenderingOpenGL-6.1.so
LIBS += /usr/local/lib/libvtkIOLSDyna-6.1.so
LIBS += /usr/local/lib/libvtkIOExport-6.1.so
LIBS += /usr/local/lib/libvtkRenderingFreeTypeOpenGL-6.1.so
LIBS += /usr/local/lib/libvtkImagingStencil-6.1.so
LIBS += /usr/local/lib/libvtkpng-6.1.so
LIBS += /usr/local/lib/libvtkFiltersFlowPaths-6.1.so
LIBS += /usr/local/lib/libvtkexoIIc-6.1.so
LIBS += /usr/local/lib/libvtkRenderingFreeType-6.1.so
LIBS += /usr/local/lib/libvtkImagingGeneral-6.1.so
LIBS += /usr/local/lib/libvtkFiltersStatistics-6.1.so
LIBS += /usr/local/lib/libvtkRenderingLabel-6.1.so
LIBS += /usr/local/lib/libvtkParallelCore-6.1.so
LIBS += /usr/local/lib/libvtkFiltersParallelImaging-6.1.so
LIBS += /usr/local/lib/libvtkverdict-6.1.so
LIBS += /usr/local/lib/libvtkCommonDataModel-6.1.so
LIBS += /usr/local/lib/libvtkIOImport-6.1.so
LIBS += /usr/local/lib/libvtkhdf5_hl-6.1.so
LIBS += /usr/local/lib/libvtkRenderingContext2D-6.1.so
LIBS += /usr/local/lib/libvtkDomainsChemistry-6.1.so
LIBS += /usr/local/lib/libvtkImagingMorphological-6.1.so
LIBS += /usr/local/lib/libvtkIOLegacy-6.1.so
LIBS += /usr/local/lib/libvtkIOXMLParser-6.1.so
LIBS += /usr/local/lib/libvtkCommonColor-6.1.so
LIBS += /usr/local/lib/libvtkalglib-6.1.so
LIBS += /usr/local/lib/libvtkViewsContext2D-6.1.so
LIBS += /usr/local/lib/libvtkGeovisCore-6.1.so
LIBS += /usr/local/lib/libvtkImagingCore-6.1.so
LIBS += /usr/local/lib/libvtkFiltersGeneral-6.1.so
LIBS += /usr/local/lib/libvtkCommonComputationalGeometry-6.1.so
LIBS += /usr/local/lib/libvtkRenderingVolumeOpenGL-6.1.so
LIBS += /usr/local/lib/libvtkoggtheora-6.1.so
LIBS += /usr/local/lib/libvtkFiltersTexture-6.1.so
LIBS += /usr/local/lib/libvtkIOPLY-6.1.so
LIBS += /usr/local/lib/libvtkfreetype-6.1.so
LIBS += /usr/local/lib/libvtkCommonCore-6.1.so
LIBS += /usr/local/lib/libvtkjpeg-6.1.so
LIBS += /usr/local/lib/libvtkFiltersGeometry-6.1.so
LIBS += /usr/local/lib/libvtkChartsCore-6.1.so
LIBS += /usr/local/lib/libvtkIOGeometry-6.1.so
LIBS += /usr/local/lib/libvtksqlite-6.1.so
LIBS += /usr/local/lib/libvtkIOMINC-6.1.so
LIBS += /usr/local/lib/libvtkFiltersSources-6.1.so
LIBS += /usr/local/lib/libvtkInteractionWidgets-6.1.so
LIBS += /usr/local/lib/libvtkzlib-6.1.so
LIBS += /usr/local/lib/libvtkIOMovie-6.1.so
LIBS += /usr/local/lib/libvtkIOXML-6.1.so
LIBS += /usr/local/lib/libvtkCommonMath-6.1.so
LIBS += /usr/local/lib/libvtkFiltersExtraction-6.1.so
LIBS += /usr/local/lib/libvtkNetCDF_cxx-6.1.so
LIBS += /usr/local/lib/libvtkFiltersCore-6.1.so
LIBS += /usr/local/lib/libvtkViewsGeovis-6.1.so
LIBS += /usr/local/lib/libvtkIOAMR-6.1.so
LIBS += /usr/local/lib/libvtkFiltersSelection-6.1.so
LIBS += /usr/local/lib/libvtkCommonExecutionModel-6.1.so
LIBS += /usr/local/lib/libvtkRenderingVolumeAMR-6.1.so
LIBS += /usr/local/lib/libvtkjsoncpp-6.1.so
LIBS += /usr/local/lib/libvtkImagingMath-6.1.so
LIBS += /usr/local/lib/libvtkIOExodus-6.1.so
LIBS += /usr/local/lib/libvtkIOInfovis-6.1.so
LIBS += /usr/local/lib/libvtkIOCore-6.1.so
LIBS += /usr/local/lib/libvtkInfovisCore-6.1.so
LIBS += /usr/local/lib/libvtkftgl-6.1.so
LIBS += /usr/local/lib/libvtkRenderingLOD-6.1.so
LIBS += /usr/local/lib/libvtkFiltersVerdict-6.1.so
LIBS += /usr/local/lib/libvtkFiltersProgrammable-6.1.so
LIBS += /usr/local/lib/libvtkIOImage-6.1.so
LIBS += /usr/local/lib/libvtkDICOMParser-6.1.so
LIBS += /usr/local/lib/libvtkRenderingImage-6.1.so
LIBS += /usr/local/lib/libvtkgl2ps-6.1.so
LIBS += /usr/local/lib/libvtkImagingSources-6.1.so
LIBS += /usr/local/lib/libvtkFiltersSMP-6.1.so
LIBS += /usr/local/lib/libvtkFiltersImaging-6.1.so
LIBS += /usr/local/lib/libvtkFiltersHyperTree-6.1.so
LIBS += /usr/local/lib/libvtkIOVideo-6.1.so
LIBS += /usr/local/lib/libvtkImagingStatistics-6.1.so
LIBS += /usr/local/lib/libvtkFiltersHybrid-6.1.so
LIBS += /usr/local/lib/libvtkInteractionImage-6.1.so
LIBS += /usr/local/lib/libvtkRenderingLIC-6.1.so
LIBS += /usr/local/lib/libvtkRenderingCore-6.1.so
LIBS += /usr/local/lib/libvtkIOParallel-6.1.so
LIBS += /usr/local/lib/libvtkFiltersGeneric-6.1.so
LIBS += /usr/local/lib/libvtkIOEnSight-6.1.so
LIBS += /usr/local/lib/libvtkCommonMisc-6.1.so

#PCL
INCLUDEPATH += /usr/local/include/pcl-1.8
LIBS += /usr/local/lib/libpcl_2d.so
LIBS += /usr/local/lib/libpcl_ml.so
LIBS += /usr/local/lib/libpcl_stereo.so
LIBS += /usr/local/lib/libpcl_common.so
LIBS += /usr/local/lib/libpcl_features.so
LIBS += /usr/local/lib/libpcl_filters.so
LIBS += /usr/local/lib/libpcl_io_ply.so
LIBS += /usr/local/lib/libpcl_io.so
LIBS += /usr/local/lib/libpcl_kdtree.so
LIBS += /usr/local/lib/libpcl_keypoints.so
LIBS += /usr/local/lib/libpcl_octree.so
LIBS += /usr/local/lib/libpcl_outofcore.so
LIBS += /usr/local/lib/libpcl_people.so
LIBS += /usr/local/lib/libpcl_recognition.so
LIBS += /usr/local/lib/libpcl_registration.so
LIBS += /usr/local/lib/libpcl_sample_consensus.so
LIBS += /usr/local/lib/libpcl_search.so
LIBS += /usr/local/lib/libpcl_segmentation.so
LIBS += /usr/local/lib/libpcl_surface.so
LIBS += /usr/local/lib/libpcl_tracking.so
LIBS += /usr/local/lib/libpcl_visualization.so\

SOURCES += main.cpp \
    Readlabels.cpp \
    proposed.cpp \
    pointcloud.cpp

HEADERS += \
    Readlabels.h \
    proposed.h \
    pointcloud.h \
    Label.h
