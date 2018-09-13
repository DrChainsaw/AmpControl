package ampcontrol.model.training.model.naming;

/**
 * Selects one out of two {@link FileNamePolicy} to use based the length of the input
 */
public class CharThreshold implements FileNamePolicy {

    private final long threshold;
    private final FileNamePolicy aboveThreshold;
    private final FileNamePolicy belowOrEqualToThreshold;

    /**
     *
     * @param threshold Threshold for what is to be considered a long name
     * @param aboveThreshold Policy to use for long names
     */
    public CharThreshold(long threshold, FileNamePolicy aboveThreshold) {
        this(threshold, aboveThreshold, identity);
    }

    /**
     *
     * @param threshold Threshold for what is to be considered a long name
     * @param aboveThreshold Policy to use for long names
     * @param belowOrEqualToThreshold Policy to use for short names
     */
    public CharThreshold(long threshold, FileNamePolicy aboveThreshold, FileNamePolicy belowOrEqualToThreshold) {
        this.threshold = threshold;
        this.aboveThreshold = aboveThreshold;
        this.belowOrEqualToThreshold = belowOrEqualToThreshold;
    }

    @Override
    public String toFileName(String str) {
        return str.length() > threshold ? aboveThreshold.toFileName(str) : belowOrEqualToThreshold.toFileName(str);
    }
}
