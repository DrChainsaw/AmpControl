package ampcontrol.model.training.data.processing;

/**
 * Info for sampling a piece of audio from a clip (e.g. a loaded file).
 *
 * @author Christian Skärby
 */
class AudioSamplingInfo {

    private final double startTime;
    private final double length;

    /**
     * Constructor
     *
     * @param startTime Start time in seconds within the clip
     * @param length    Lenght of sample in seconds w.r.t start time
     */
    AudioSamplingInfo(double startTime, double length) {
        this.startTime = startTime;
        this.length = length;
    }

    /**
     * Returns the start time of the sample in seconds
     *
     * @return the start time of the sample in seconds
     */
    double getStartTime() {
        return startTime;
    }

    /**
     * Returns the length of the sample in seconds
     *
     * @return the length of the sample in seconds
     */
    double getLength() {
        return length;
    }
}
