package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Divides epoch by a given step, practically causing epochs in an underlying {@link ISchedule} to appear to increase
 * a factor step slower. Design is most likely the result of serialization-means-no-refactoring induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Step implements ISchedule {

    private final int step;
    private final ISchedule source;

    public Step(
            @JsonProperty("step") int step,
            @JsonProperty("source") ISchedule source) {
        this.step = step;
        this.source = source;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return source.valueAt(iteration, epoch / step);
    }

    @Override
    public ISchedule clone() {
        return new Step(step, source.clone());
    }
}
