vtune -collect gpu-hotspots -- ./LLM_SYCL -b 1 -g 10 -x -y -s
vtune -report hotspots -group-by=computing-instance -format=csv > llm.sycl.vtune.csv