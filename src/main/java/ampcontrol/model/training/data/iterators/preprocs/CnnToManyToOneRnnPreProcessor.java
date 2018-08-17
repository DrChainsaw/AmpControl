package ampcontrol.model.training.data.iterators.preprocs;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * {@link DataSetPreProcessor} which transforms CNN (4D) {@link DataSet DataSets} to RNN (3D) {@link DataSet DataSets}.
 * Adds masking so that only last time step of RNN is expected to have a label (i.e many-to-one output).
 * <br><br>
 * Transforms features of shape [a,b,c,d] to shape [a,d,c] basically assuming that 1) input has only a single channel
 * (b == 1) and 2) c is the dimension which is relevant to view as time.
 * <br><br>
 * Transforms labels of shape [a,x] to shape [a,x,c]
 * <br><br>
 * Adds a label mask of shape [a,c] which is 1 for time index c-1 and 0 otherwise.
 *
 * @author Christian Skärby
 */
public class CnnToManyToOneRnnPreProcessor implements DataSetPreProcessor {


    /**
	 * 
	 */
	private static final long serialVersionUID = -5704355643721805497L;

	@Override
    public synchronized void preProcess(DataSet toPreProcess) {

        INDArray oldLabels = toPreProcess.getLabels().dup();
      //  System.out.println("OldLabs: \\n" + oldLabels);
        long[] labelsShape = oldLabels.shape();
        long[] featuresShape = toPreProcess.getFeatures().shape();

        INDArray newLabels = Nd4j.create(new long[]{labelsShape[0], labelsShape[1], featuresShape[2]});
        INDArray labelMask = Nd4j.create(new long[]{labelsShape[0], featuresShape[2]});
        INDArrayIndex[] lastTimeStep = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(featuresShape[2]-1)};
        newLabels.put(lastTimeStep, oldLabels);
        lastTimeStep = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(featuresShape[2]-1)};
        labelMask.put(lastTimeStep, 1);

        INDArray newFeatures = cnnToRnnFeature(toPreProcess.getFeatures().dup());
        toPreProcess.setFeatures(newFeatures);
        toPreProcess.setLabels(newLabels);
        toPreProcess.setLabelsMaskArray(labelMask);
    }

    /**
     * Utility function for (crudely) transforming CNN (4D) features to RNN (3D) features. Transforms input of shape
     * [a,b,c,d] to shape [a, d, c] basically assuming that 1) input has only a single channel (b == 1) and 2) c is the
     * dimension which is relevant to view as time.
     *
     * @param cnnFeatures 4D CNN input
     * @return 3D RNN input
     */
	public static INDArray cnnToRnnFeature(INDArray cnnFeatures) {
	    if(cnnFeatures.size(1) > 1) {
	        // Implementation could probably be reworked to do something like [miniBatchSize, width*channels, height]
            throw new IllegalArgumentException("Hacky implementation assumes single channel only. Got " + cnnFeatures.size(1));
        }
		INDArray rnnFeatures = cnnFeatures.tensorAlongDimension(0, 0, 3, 2);
        rnnFeatures = rnnFeatures.swapAxes(1,2);
        //int[] shape = rnnFeatures.shape();
        return rnnFeatures;
	}
}
