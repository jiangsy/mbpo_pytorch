for env in "halfcheetah" "walker2d" "hopper" "ant"
do
  python run_mbpo.py --configs "mbpo.yaml" "${env}.yaml" "priv.yaml"
done