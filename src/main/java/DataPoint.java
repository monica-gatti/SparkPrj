    import java.io.Serializable;
    public class DataPoint implements Serializable {
        private double x;
        private double y;
        // Altri campi e metodi...
        public void setY(double parseDouble) {
        }
        public void setX(double parseDouble) {
            this.x = parseDouble;
        }
        public Double getX() {
            return this.x;
        }
        public Double getY() {
            return this.y;
        }
      }
      
      
