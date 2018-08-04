package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Makes a given {@link ISchedule} cyclic. Design is most likely the result of serialization-means-no-refactoring induced
 * insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Cyclic implements ISchedule {

    private final int period;
    private final ISchedule source;

    public Cyclic(
            @JsonProperty("period") int period,
            @JsonProperty("source") ISchedule source) {
        this.period = period;
        this.source = source;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return source.valueAt(iteration, epoch % period);
    }

    @Override
    public ISchedule clone() {
        return new Cyclic(period, source.clone());
    }
}
