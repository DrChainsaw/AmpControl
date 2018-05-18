package ampcontrol.model.training.data;

import ampcontrol.audio.processing.ProcessingResult;

import java.util.stream.Stream;

/**
 * Interface for streaming {@link TrainingData}.
 *
 * @author Christian Skärby
 */
public interface DataProvider {
    Stream<DataProvider.TrainingData> generateData();

    /**
     * Hold data needed for training. {@link ProcessingResult} holds numerical features
     * and label is the literal label for the features.
     */
    class TrainingData {
        private final String label;
        private final ProcessingResult result;
        public TrainingData(String label, ProcessingResult result) {
            this.label = label;
            this.result= result;
        }

        /**
         * Returns the label
         * @return the label
         */
        public String getLabel() {
            return label;
        }

        /**
         * Returns the {@link ProcessingResult}
         * @return the {@link ProcessingResult}
         */
        public ProcessingResult result() {
            return result;
        }
    }
}
