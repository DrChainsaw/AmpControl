package ampcontrol.audio;

/**
 * Represents buffered audio input.
 *
 * @author Christian Skärby
 */
public interface AudioInputBuffer {

    /**
     * Returns the contents of the buffer
     *
     * @return the contents of the buffer
     */
    double[] getAudio();
}
