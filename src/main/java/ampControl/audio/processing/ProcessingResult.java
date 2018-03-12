package ampControl.audio.processing;

import java.util.List;

/**
 * Result from input processing
 *
 * @author Christian Skärby
 */
public interface ProcessingResult {

    /**
     * The processing to create a ProcessingResult
     */
    interface Processing extends ProcessingResult {

        /**
         * Receive input to process
         *
         * @param input
         */
        void receive(double[][] input);

        /**
         * Return the name of this post processing.
         *
         * @return name of this post processing.
         */
        String name();
    }

    /**
     * Return the result. Example of when the returned {@link List} has more than one element is when processing is
     * {@link Fork forked}.
     *
     * @return the result
     */
    List<double[][]> get();
}
