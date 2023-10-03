To reproduce the enterprise credit assessment results on CED, you can implement following code:

    python train.py
To run the machine learning methods on CED, you can implement following code:

    python ml_baselines.py

To reproduce the node classification results on DBLP, you need to firstly run:

    python process_dblp.py
    python ./DBLP/data/metapath.py

 afterwards, you can implement following code:

    python ./DBLP/train.py

To reproduce the DistShift results on DBLP, you can implement following code:

    python distshift.py
