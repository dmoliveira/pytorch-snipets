#!/usr/bin/env bash
if [ ! -d "myenv" ]; then
  python3 -m venv myenv
  source ./myenv/bin/activate
  pip install -U "numpy<2" \
              torch \
              torchvision \
              scikit-learn \
              pandas \
              plotly
else
   source ./myenv/bin/activate
fi


