package ampcontrol.model.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Represents a Classifier i.e the model which turns input (e.g sound) into a set of probabilities for classes,
 * e.g. current sound is 10% chance of being rythm, 90% chance of being lead.
 *
 * @author Christian Skärby
 */
public interface Classifier {

    /**
     * Returns probabilities of each class as an {@link INDArray}.
     *
     * @return probabilities of each class as an {@link INDArray}.
     */
    INDArray classify();

    /**
     * Returns the estimated accuracy of the model.
     *
     * @return
     */
    double getAccuracy();
    
}
