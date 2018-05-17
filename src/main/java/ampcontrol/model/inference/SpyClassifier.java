package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.model.visualize.PlotSpectrogram;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * "Spy" {@link Classifier} intended for online accuracy trouble-shooting of a provided {@link Classifier}. Is transparent
 * w.r.t. output of the source classifier but has a configured "right" and "wrong" classification. If more than maxAccum
 * "correct" classifications and more than maxAccum "incorrect" classifications has been made by the source classifier
 * then the given {@link Listener Listeners} process method will be invoked so that both types of input can be examined
 * or stored.
 * <br><br>
 * Intended use is to keep feeding input(e.g. by playing it on the instrument while plugged in and program is running)
 * of the "right" type which one has observed to often be incorrectly classified and e.g. plot what the input looks like
 * when misclassification happens.
 *
 * @author Christian Skärby
 */
class SpyClassifier implements Classifier {

    private final Classifier sourceClassifier;
    private final ClassifierInputProvider sourceInput;
    private final int maxAccum;
    private final int right;
    private final int wrong;
    private final double threshold = 0.8;

    private boolean tic = true;
    private int nrofRight = 0;
    private int nrofWrong = 0;
    private INDArray correctlyClassifiedInput;
    private INDArray incorrectlyClassifiedInput;
    private Runnable reportResult;

    /**
     * Listener for correctly and incorrectly classified input respectively
     */
    interface Listener {
        /**
         * Process the input.
         * @param correctlyClassifiedInput
         * @param incorrectlyClassifiedInput
         */
        void process(INDArray correctlyClassifiedInput, INDArray incorrectlyClassifiedInput);
    }

    /**
     * Plots the spectrograms which are correctly and incorrectly classified in two separate plots.
     */
    static class PlotSpecgramListener implements Listener {
        @Override
        public void process(INDArray correctlyClassifiedInput, INDArray incorrectlyClassifiedInput) {
            if(correctlyClassifiedInput != null) {
                PlotSpectrogram.plot(correctlyClassifiedInput, 0, 1);
            }
            if(incorrectlyClassifiedInput != null) {
                PlotSpectrogram.plot(incorrectlyClassifiedInput,0,1);
            }
        }
    }

    /**
     * Stores the reported inputs.
     */
    static class StoringListener implements Listener {

        private INDArray correctlyClassifiedInput;
        private INDArray incorrectlyClassifiedInput;
        private boolean hasInput = false;

        @Override
        public void process(INDArray correctlyClassifiedInput, INDArray incorrectlyClassifiedInput) {
            hasInput = true;
            this.correctlyClassifiedInput = correctlyClassifiedInput;
            this.incorrectlyClassifiedInput = incorrectlyClassifiedInput;
        }

        /**
         * Returns true if the listener has gotten any input
         * @return true if the listener has gotten any input
         */
        public boolean hasInput() {
            return hasInput;
        }

        /**
         * Gets the correctly classified input
         * @return the correctly classified input
         */
        public INDArray getCorrectlyClassifiedInput() {
            return correctlyClassifiedInput;
        }

        /**
         * Gets the incorrectly classified input
         * @return the incorrectly classified input
         */
        public INDArray getIncorrectlyClassifiedInput() {
            return incorrectlyClassifiedInput;
        }
    }

    /**
     * Constructor
     * @param sourceClassifier Classifier to spy on
     * @param sourceInput Input to report
     * @param listener Listens to reported input
     * @param maxAccum How many sequential correct respectively incorrect classifications before reporting
     * @param right "Correct" label
     * @param wrong "Incorrect" label
     */
    SpyClassifier(
            Classifier sourceClassifier,
            ClassifierInputProvider sourceInput,
            Listener listener,
            int maxAccum,
            int right,
            int wrong) {
        this.sourceClassifier = sourceClassifier;
        this.sourceInput = sourceInput;
        this.maxAccum = maxAccum;
        this.right = right;
        this.wrong = wrong;

        reportResult = () -> listener.process(correctlyClassifiedInput, incorrectlyClassifiedInput);
    }

    @Override
    public INDArray classify() {
        INDArray classification = sourceClassifier.classify();
        accumInput(classification);
        return classification;
    }

    private void accumInput(INDArray classification) {
        int highestProb = classification.argMax(1).getInt(0);

        if(classification.getDouble(highestProb) > threshold) {
            if(highestProb == right) {
                boolean reset = !tic ||correctlyClassifiedInput == null;
                if(reset) {
                    nrofRight = 0;
                }
                if(nrofRight < maxAccum) {
                    correctlyClassifiedInput = accumulate(reset, correctlyClassifiedInput, sourceInput.getModelInput());
                    nrofRight++;
                }
                tic = true;
            }
            if(highestProb == wrong) {
                boolean reset = tic || incorrectlyClassifiedInput == null;
                if(reset) {
                    nrofWrong = 0;
                }
                if(nrofWrong < maxAccum) {
                    incorrectlyClassifiedInput = accumulate(reset, incorrectlyClassifiedInput, sourceInput.getModelInput());
                    nrofWrong++;
                }
                tic = false;
            }

            if(nrofRight + nrofWrong == 2*maxAccum) {
                reportResult.run();
                reportResult = () -> {};
            }
        }
    }

    private INDArray accumulate(boolean shallReset, INDArray toAccum, INDArray input) {
        //int[] shape = input.shape();
        if(shallReset) {
            toAccum = input.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        } else {
            toAccum = Nd4j.vstack(toAccum, input.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()));
        }
        return toAccum;
    }

	@Override
	public double getAccuracy() {
		return sourceClassifier.getAccuracy();
	}
}
