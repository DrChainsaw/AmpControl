package ampcontrol.amp;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface to receive a classification in the form of a vector where each element corresponds to the estimated
 * probability of the associated label (e.g. Rythm, Lead, Cat, etc.).
 *
 * @author Christian Skärby
 */
public interface ClassificationListener {

    /**
     * Factory interface
     */
    interface Factory {
        /**
         * Create a {@link ClassificationListener}.
         *
         * @return
         */
        ClassificationListener create();
    }

    /**
     * Indicate label probabilities
     *
     * @param probabilities
     */
    void indicateAudioClassification(INDArray probabilities) ;

}
