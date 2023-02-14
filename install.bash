git submodule init
git submodule update
cd build
if [ -z $1 ]; then
        echo "Using default values of STATE_SIZE, CONTROL_SIZE, KNOT_POINTS"
        cmake .. -DS=14 -DC=7 -DK=50
elif [ -z $2 ]; then
        echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS"
        cd ..
        return
elif [ -z $3 ]; then
        echo "Not enough arguments, please specify STATE_SIZE, CONTROL_SIZE, KNOT_POINTS"
        cd ..
        return
else
        cmake .. -DS=$1 -DC=$2 -DK=$3
        
fi
make
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..