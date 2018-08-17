package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Value is epoch times a fixed coefficient. Design is most likely the result of serialization-means-no-refactoring
 * induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Linear implements ISchedule {

    private final double coefficient;

    public Linear(
            @JsonProperty("coefficient") double coefficient) {
        this.coefficient = coefficient;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return epoch * coefficient;
    }

    @Override
    public ISchedule clone() {
        return new Linear(coefficient);
    }
}
