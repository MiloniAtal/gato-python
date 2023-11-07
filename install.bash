git submodule init
git submodule update
cd build
if [ -z $1 ]; then
        echo "Using default values of STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, NC, NX"
        cmake .. -DS=3 -DC=3 -DK=3 -DNC=9 -DNX=9
elif [ -z $2 ]; then
        echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, NC, NX"
        cd ..
        return
elif [ -z $3 ]; then
       echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, NC, NX"
        cd ..
        return
elif [ -z $4 ]; then
        echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, NC, NX"
        cd ..
        return
elif [ -z $5 ]; then
        echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, NC, NX"
        cd ..
        return
else
        cmake .. -DS=$1 -DC=$2 -DK=$3 -DNC=$4 -DNX=$5
        
fi
make
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..