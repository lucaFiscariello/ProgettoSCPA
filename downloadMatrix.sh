#!/bin/bash 

arrayFileName=( cage4 mhda416 mcfe olm1000 adder_dcop_32 west2021 cavity10 rdist2 cant olafu Cube_Coup_dt0 ML_Laplace bcsstk17 mac_econ_fwd500 mhd4800a cop20k_A raefsky2 af23560 lung2 PR02R FEM_3D_thermal1 thermal1 thermal2 thermomech_TK nlpkkt80 webbase-1M dc1 amazon0302 af_1_k101 roadNet-PA )
arrayDowloadLink=( 
    https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage4.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhda416.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Bai/olm1000.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Sandia/adder_dcop_32.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/HB/west2021.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity10.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Zitney/rdist2.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Simon/olafu.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Laplace.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhd4800a.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Simon/raefsky2.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Norris/lung2.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/PR02R.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal1.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt80.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/dc1.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_1_k101.tar.gz
    https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz
    )
    
length=${#arrayFileName[@]}


for (( j=0; j<length; j++ ));
do
    wget ${arrayDowloadLink[$j]}
    tar -zxvf ${arrayFileName[$j]}.tar.gz 
    rm ${arrayFileName[$j]}.tar.gz 
    mv ${arrayFileName[$j]}/${arrayFileName[$j]}.mtx matrixFile
    rm -r ${arrayFileName[$j]}
done


