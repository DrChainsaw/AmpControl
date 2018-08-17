package ampcontrol.model.training.schedule;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Limits value from an {@link ISchedule} to a maximum value. Design is most likely the result of
 * serialization-means-no-refactoring induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class MaxLim implements ISchedule {

    private final double limit;
    private final ISchedule source;

    public MaxLim(
            @JsonProperty("limit") double limit,
            @JsonProperty("source") ISchedule source) {
        this.limit = limit;
        this.source = source;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return Math.min(limit, source.valueAt(iteration, epoch));
    }

    @Override
    public ISchedule clone() {
        return new MaxLim(limit, source.clone());
    }
}
