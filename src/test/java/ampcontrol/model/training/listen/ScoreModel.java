package ampcontrol.model.training.listen;

final class ScoreModel extends MockModel {
    private final double score;

    public ScoreModel(double score) {
        this.score = score;
    }

    @Override
    public double score() {
        return score;
    }
}
