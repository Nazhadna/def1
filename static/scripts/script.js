var canvas = document.getElementById("paint");
var ctx = canvas.getContext("2d");
ctx.font = "22px Verdana";
var width = $("/static/uploads/image.jpg").width();
var height = $("/static/uploads/image.jpg").height();
var hold = false;
ctx.lineWidth = 2;
ctx.strokeStyle='#FF00000';
ctx.fillStyle='#FF0000';
var fill_value = true;
var stroke_value = false;
var brushRadius = 8;

var colors = ['#0000FF', '#FF0000', '#00FF00', '#FF00FF'];

                        
function color(color_value){
    ctx.strokeStyle = color_value;
    ctx.fillStyle = color_value;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

}    
        
function add_pixel(){
    ctx.lineWidth += 1;
}
        
function reduce_pixel(){
    if (ctx.lineWidth == 1){
        ctx.lineWidth = 1;
    }
    else{
        ctx.lineWidth -= 1;
    }
}
        
function fill(){
    fill_value = true;
    stroke_value = false;
}
        
function outline(){
    fill_value = false;
    stroke_value = true;
}


function pencil(){

    let curves = [];

    function makePoint(x, y) {
        return [x, y];
    };

    canvas.addEventListener("mousedown", event => {
        const curve = [];
        curve.color = ctx.strokeStyle;
        curve.lineWidth = ctx.lineWidth;
        curve.push(makePoint(event.offsetX, event.offsetY));
        curves.push(curve);
        hold = true;
    });

    canvas.addEventListener("mousemove", event => {
        if(hold){
            const point = makePoint(event.offsetX, event.offsetY)
            curves[curves.length - 1].push(point);
            repaint(curves);
        }
    });

    canvas.addEventListener("mouseup", event => {
        hold = false;
    });

    canvas.addEventListener("mouseleave", event => {
        hold = false;
    });
}

function repaint(curves) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    curves.forEach((curve) => {
        if (ctx.strokeStyle == curve.color) {
            ctx.lineWidth = curve.lineWidth;
            circle(curve[0]);
            smoothCurve(curve);
        }
    });
}

function circle(point) {
    ctx.beginPath();
    ctx.arc(...point, brushRadius / 2, 0, 2 * Math.PI);
    ctx.fill();
}

function smoothCurveBetween (p1, p2) {
    const cp = p1.map((coord, idx) => (coord + p2[idx]) / 2);
    ctx.quadraticCurveTo(...p1, ...cp);
}

function smoothCurve(points) {
    ctx.beginPath();
    ctx.lineWidth = brushRadius;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.moveTo(...points[0]);

    for(let i = 1; i < points.length - 1; i++) {
        smoothCurveBetween(points[i], points[i + 1]);
    }

    ctx.stroke();
}

function save(){
/*    var filename = document.getElementById("fdbcktype").value;
    var data = JSON.stringify(brushRadius);
    var image = canvas.toDataURL();
    
    $.post("/", { save_fname: filename, save_cdata: data, save_image: image });*/

    colors.forEach((color) => {
        ctx.strokeStyle = color;
        repaint();
        var image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
        var a = document.createElement('a');
        a.href = image;
        a.download = 'my-canvas.jpeg';
        document.body.appendChild(a);
        a.click();
    });
    /*
    var image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
    var a = document.createElement('a');
    a.href = image;
    a.download = 'my-canvas.jpeg';
    document.body.appendChild(a);
    a.click();*/
}
