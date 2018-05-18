package ampcontrol.amp.labelmapping;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic {@link LabelMapping} which masks mappings until momentumThreshold number of identical labels have been
 * received. Acts as a low pass filter, reducing variance at the cost of delays.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class MomentumLabelMapping<T> implements LabelMapping<T> {

    private final LabelMapping<T> sourceLabelMapping;
    private final int momentumThreshold;

    private int momentumCount = 0;
    private int lastLabel;

    public MomentumLabelMapping(int momentumThreshold, LabelMapping<T> sourceLabelMapping) {
        this.sourceLabelMapping = sourceLabelMapping;
        this.momentumThreshold = momentumThreshold;
    }

    @Override
    public List<T> apply(Integer labelInd) {
        if(lastLabel != labelInd) {
            lastLabel = labelInd;
            momentumCount = 0;
        }
        momentumCount++;
        if(momentumCount < momentumThreshold) {
            return new ArrayList<>();
        }
        return sourceLabelMapping.apply(labelInd);

    }
}
