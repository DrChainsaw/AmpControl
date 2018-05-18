package ampcontrol.audio.processing;

import ampcontrol.audio.processing.ProcessingResult.Factory;

/**
 * Connects two {@link Factory} so that output from the first is input to the second
 *
 * @author Christian Skärby
 */
public class Pipe implements Factory {

    private final Factory first;
    private final Factory second;

    public Pipe(Factory first, Factory second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return second.create(first.create(input));
    }

    @Override
    public String name() {
        return first.name() + nameStatic() + second.name();
    }

    public static String nameStatic() {
        return "_pipe_";
    }

}
