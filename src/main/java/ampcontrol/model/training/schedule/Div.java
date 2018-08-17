package ampcontrol.model.training.schedule;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Divide the output from one {@link ISchedule} with another. Design is most likely the result of
 * serialization-means-no-refactoring induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Div implements ISchedule {
    private final ISchedule first;
    private final ISchedule second;

    public Div(
            @JsonProperty("first") ISchedule first,
            @JsonProperty("second") ISchedule second) {
        this.first = first;
        this.second = second;
    }


    @Override
    public double valueAt(int iteration, int epoch) {
        return first.valueAt(iteration,epoch) / second.valueAt(iteration, epoch);

    }

    @Override
    public ISchedule clone() {
        return new Div(first.clone(), second.clone());
    }
}
