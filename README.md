To reproduce the enterprise credit assessment results on ECAD, you can implement following code:

    python train.py


To run the machine learning methods on ECAD, you can implement following code:

    python ml_baselines.py

To reproduce the results on SMEsD dataset, you can run:

    cd ./SMEsD/
    python train.py

To reproduce the node classification results on DBLP, you need to firstly run:

    python process_dblp.py
    python ./DBLP/data/metapath.py

 afterwards, you can implement following code:

    cd ./DBLP/
    python train.py

To reproduce the DistShift results on DBLP, you can implement following code:

    python distshift.py
