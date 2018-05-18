package ampcontrol.audio.processing;

import java.util.List;
import java.util.stream.Stream;

/**
 * Result from input processing
 *
 * @author Christian Skärby
 */
public interface ProcessingResult {

    /**
     * The factory to create a ProcessingResult
     */
    interface Factory {

        /**
         * Return the name of this post processing.
         *
         * @return name of this post processing.
         */
        String name();

        /**
         * Receive input to process
         *
         * @param input
         */
        ProcessingResult create(ProcessingResult input);

    }

    /**
     * Return the result. Example of when the returned {@link List} has more than one element is when processing is
     * {@link Fork forked}.
     *
     * @return the result
     */
    Stream<double[][]> stream();
}
