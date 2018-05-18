package ampcontrol.audio;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Provides input for classifiction as an INDArray.
 *
 * @author Christian Skärby
 */
public interface ClassifierInputProvider {

    /**
     * Handle for updating a {@link ClassifierInputProvider}
     */
    interface UpdateHandle {
        /**
         * Updates the input, e.g. read new samples from an input buffer.
         */
        void updateInput();
    }

    /**
     * Union of {@link ClassifierInputProvider} and {@link Updatable}. An updatable input provider simply put
     */
    interface Updatable extends UpdateHandle, ClassifierInputProvider {

    }

    /**
     * Returns input for a classifier
     *
     * @return input for a classifier
     */
    INDArray getModelInput();
}
