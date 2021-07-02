#!/bin/bash

#set defaults

qos='medium'
cpus=4

scenario_name=''
workdir_root='/p/tmp/robinmid/acclimate_run/scaling_ensembles'
acclimate_path='/home/robinmid/repos/acclimate/acclimate/build/acclimate'
ensembles_dir='/home/robinmid/repos/harvey_scaling/data/forcing/ensembles'

#check args
while :; do
  case $1 in
  --qos)
    if [ "$2" ]; then
      qos=$2
      shift
    fi
    ;;
  --scenario)
    if [ "$2" ]; then
      scenario_name=$2
      shift
    fi
    ;;
  --cpus)
    if [ "$2" ]; then
      cpus=$2
      shift
    fi
    ;;
  *)
    break
    ;;
  esac
  shift
done

if [ "$scenario_name" = "" ]; then
  echo "Need to set a settings folder name for the scenario to choose! Exiting..."
  exit 1
fi

timestamp="$(date +%F_%T)"
workdir_root="${workdir_root}/${scenario_name}/${timestamp}"
mkdir -p "$workdir_root"

echo "Simulating acclimate with settings files for scenario ${scenario_name}."

cp "${ensembles_dir}/${scenario_name}/ensemble_meta.pk" ${workdir_root}

for filepath in ${ensembles_dir}/${scenario_name}/settings_*.yml; do
    filename=${filepath##*/}
    iter_dir="$workdir_root/${filename:9:-4}"
    mkdir ${iter_dir}
    cp ${filepath} ${iter_dir}/
    echo "Submitting acclimate simluation with settings file ${filepath}"
    sh ~/scripts/run_cluster.sh '--job_name' "${filename:9:-4}" '--qos' "$qos" '--dir' "$iter_dir" '--cpus' "$cpus" "$acclimate_path $iter_dir/$filename"
done
