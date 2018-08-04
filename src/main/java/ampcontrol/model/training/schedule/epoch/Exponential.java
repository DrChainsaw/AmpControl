package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Value is base^epoch. Design is most likely the result of serialization-means-no-refactoring induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Exponential implements ISchedule {

    private final double base;

    public Exponential(
            @JsonProperty("base") double base) {
        this.base = base;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return Math.pow(base, epoch);
    }

    @Override
    public ISchedule clone() {
        return new Exponential(base);
    }
}
