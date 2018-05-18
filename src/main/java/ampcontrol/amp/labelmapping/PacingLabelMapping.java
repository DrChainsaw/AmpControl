package ampcontrol.amp.labelmapping;

import java.time.Duration;
import java.time.LocalTime;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link LabelMapping} which paces messages so that two messages are
 * never sent with a shorter interval than the one specified here
 *
 * @author Christian Skärby
 */
public class PacingLabelMapping<T> implements LabelMapping<T> {

    private final long minimumIntervalMs;
    private final LabelMapping<T> sourceLabelMapping;

    private LocalTime lastRequest;

    public PacingLabelMapping(long minimumIntervalMs, LabelMapping<T> sourceLabelMapping) {
        this.minimumIntervalMs = minimumIntervalMs;
        this.sourceLabelMapping = sourceLabelMapping;
        lastRequest = LocalTime.now().minus(2*minimumIntervalMs, ChronoField.MILLI_OF_DAY.getBaseUnit());
    }

    @Override
    public List<T> apply(Integer labelInd) {

        if(minimumIntervalMs > 0) {
            LocalTime toCompare = LocalTime.now().minus(minimumIntervalMs, ChronoField.MILLI_OF_DAY.getBaseUnit());
            Duration timeRemaining = Duration.between(lastRequest, toCompare);
            if (timeRemaining.isNegative() || timeRemaining.isZero()) {
                return new ArrayList<>();
            }
        }

        List<T> ret =  sourceLabelMapping.apply(labelInd);
        if(!ret.isEmpty()) {
            lastRequest = LocalTime.now();
        }
        return ret;


    }
}
