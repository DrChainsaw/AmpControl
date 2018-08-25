package ampcontrol.audio.processing;

import org.jetbrains.annotations.Nullable;

/**
 * Creates a {@link ProcessingResult.Factory} based on a name string. Used for persistence since
 * all preprocessing needs be recreated when restoring a saved model.
 * <br><br>
 * This class, like much else in this project is the result of passing time while training.
 * TODO Probably would have been wiser to have a serializable factory instead...
 *
 * @author Christian Skärby
 */
public class ProcessingFactoryFromString {

    private final double samplingFreq;
    private final static String prefix = "_sgpp_";

    public ProcessingFactoryFromString(double samplingFreq) {
        this.samplingFreq = samplingFreq;
    }

    public ProcessingResult.Factory get(String nameStr) {

        String preForkStr = nameStr;
        if (nameStr.matches(matchStr(".*" + Fork.matchStrStatic()))) {
            ProcessingResult.Factory forkSupplier;
            String[] forkStrs = Fork.splitFirst(nameStr);
            preForkStr = forkStrs[0];
            if (forkStrs.length == 2) {
                final String forkStr = forkStrs[1];
                final String[] endSplit = Fork.splitEnd(forkStr);
                final String pathStr = endSplit[0];
                final String[] paths = Fork.splitMid(pathStr);
                final ProcessingResult.Factory finalFirst = get(prefix() + paths[0]);
                final ProcessingResult.Factory finalSecond = get(prefix() + paths[1]);
                forkSupplier = new Fork(finalFirst, finalSecond);

                if (endSplit.length == 2 && endSplit[1].contains(Pipe.nameStatic())) {
                    forkSupplier = getPipedSupplier(endSplit[1], forkSupplier, null);
                }

                if (preForkStr.replace(prefix(), "").isEmpty()) {
                    return forkSupplier;
                }
                return getPipedSupplier(preForkStr, null, forkSupplier);
            }
        }

        if (preForkStr.matches(matchStr(".*" + Pipe.nameStatic()))) {
            return getPipedSupplier(preForkStr, null, null);
        }

        if (preForkStr.matches(matchStr(NoProcessing.nameStatic()))) {
            return new NoProcessing();
        }

        if (preForkStr.matches(matchStr(UnitMaxZeroMin.nameStatic()))) {
            return new UnitMaxZeroMin();
        }


        if (preForkStr.matches(matchStr(LogScale.nameStatic()))) {
            return new LogScale();
        }

        if (preForkStr.matches(matchStr(UnitStdZeroMean.nameStatic()))) {
            return new UnitStdZeroMean();
        }

        if (preForkStr.matches(matchStr(Mfsc.nameStatic()))) {
            return new Mfsc(samplingFreq);
        }

        if (preForkStr.matches(matchStr(Dct.nameStatic()))) {
            return new Dct();
        }

        if (preForkStr.matches(matchStr(Spectrogram.nameStatic()))) {
            return new Spectrogram(preForkStr);
        }

        if (preForkStr.matches(matchStr(ZeroMean.nameStatic()))) {
            return new ZeroMean();
        }

        if (preForkStr.matches(matchStr(Ycoord.nameStatic()))) {
            return new Ycoord();
        }

        if (preForkStr.matches(matchStr(Log10.nameStatic()))) {
            return new Log10();
        }

        if (preForkStr.matches(matchStr(Buffer.nameStatic()))) {
            return new Buffer();
        }

        return new UnitMaxZeroMean();
    }

    @Nullable
    private ProcessingResult.Factory getPipedSupplier(
            String nameStr,
            ProcessingResult.Factory first,
            ProcessingResult.Factory last) {
        ProcessingResult.Factory ret = null;
        for (String pipeStr : nameStr.split(Pipe.nameStatic())) {
            if (!pipeStr.isEmpty()) {
                if (first == null) {
                    // System.out.println("Ask first pipe: " + pipeStr);
                    ret = get(pipeStr);
                    first = ret;
                } else {
                    // System.out.println("do pipe for: " + first.get().name() + " and " + pipeStr);
                    final ProcessingResult.Factory second = get(prefix() + pipeStr);
                    // System.out.println("Got second: " + finalSecond.get().name());
                    ret = new Pipe(first, second);
                    first = ret;
                }
            }
        }

        if (last != null) {
            ret = new Pipe(ret, last);
        }
        return ret;
    }

    public static String prefix() {
        return prefix;
    }

    private static String matchStr(String aStr) {
        return ".*" + prefix() + aStr + ".*";
    }

}
