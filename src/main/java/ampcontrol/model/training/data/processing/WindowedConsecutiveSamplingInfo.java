package ampcontrol.model.training.data.processing;

import ampcontrol.model.training.data.state.SimpleStateFactory;
import ampcontrol.model.training.data.state.StateFactory;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Maps a {@link Path} to {@link AudioSamplingInfo} in a manner so that consecutive time windows are provided for each
 * unique {@link Path} given. Does not require that the same {@link Path} is provided in consecutive calls.
 * <br><br>
 * Example with clipLength = 1000 ms, windowSize = 100 ms:
 * path0 -> start @ 0 ms
 * path1 -> start @ 0 ms
 * path0 -> start @ 100 ms
 * path2 -> start @ 0 ms
 * path0 -> start @ 200 ms
 * path1 -> start @ 100 ms
 * ...
 * pathN -> start @ 900 ms
 * ...
 * pathN -> start @ 0 ms
 * etc.
 *
 * @author Christian Skärby
 */
public class WindowedConsecutiveSamplingInfo implements Function<Path, AudioSamplingInfo> {

    private final SimpleListNode<AudioSamplingInfo> samplingInfoSequence;
    private final Supplier<Map<Path, SimpleListNode<AudioSamplingInfo>>> infoMap;

    // Pretty primitive stuff. I could not find it in any of the dependent libraries....
    private static class SimpleListNode<T> {
        private SimpleListNode<T> next;
        private final T data;

        public SimpleListNode(T data) {
            this.data = data;
        }

        private void setNext(SimpleListNode<T> next) {
            this.next = next;
        }

        private SimpleListNode<T> getNext() {
            return next;
        }

        private T getData() {
            return data;
        }
    }


    /**
     * Constructor
     *
     * @param clipLengthMs (Assumed) length of clips in milliseconds
     * @param windowSizeMs Wanted window size in milliseconds
     * @param stateFactory Used to create the state of this class
     */
    public WindowedConsecutiveSamplingInfo(int clipLengthMs, int windowSizeMs, StateFactory stateFactory) {

        infoMap = stateFactory.createNewStateReference(HashMap::new, new HashMap<>());
        // Create a circular linked list to be used by all files. Each file just keeps track of its own position in the list
        samplingInfoSequence = new SimpleListNode<>(new AudioSamplingInfo(0, ms2s(windowSizeMs)));
        SimpleListNode<AudioSamplingInfo> prev = samplingInfoSequence;
        final int nrofSamplesToCreate = clipLengthMs / windowSizeMs;
        for (int winInd = 1; winInd < nrofSamplesToCreate; winInd++) {
            prev.setNext(new SimpleListNode<>(
                    new AudioSamplingInfo(ms2s(windowSizeMs * winInd), ms2s(windowSizeMs))));
            prev = prev.getNext();
        }
        prev.setNext(samplingInfoSequence);
    }

    private static double ms2s(int msVal) {
        return msVal / 1000d;
    }

    @Override
    public synchronized AudioSamplingInfo apply(Path file) {
        Map<Path, SimpleListNode<AudioSamplingInfo>> state = infoMap.get();
        state.putIfAbsent(file, samplingInfoSequence);
        SimpleListNode<AudioSamplingInfo> node = state.get(file);
        state.put(file, node.getNext());
        return node.getData();
    }

    public static void main(String[] args) {
        WindowedConsecutiveSamplingInfo info = new WindowedConsecutiveSamplingInfo(1000, 100, new SimpleStateFactory(123));
        Path f = Paths.get("aaa");
        AudioSamplingInfo aInfo = info.apply(f);
        System.out.println("start: " + aInfo.getStartTime() + " len: " + aInfo.getLength());
        for (int i = 0; i < 15; i++) {
            aInfo = info.apply(f);
            System.out.println("start: " + aInfo.getStartTime() + " len: " + aInfo.getLength());
        }
    }
}
