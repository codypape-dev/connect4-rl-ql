const cols = 7;
const rows = 6;
const cellSize = 100;
const diameter = 80;
const board = Array(rows)
  .fill()
  .map(() => Array(cols).fill(0));
let player = 1;
let playerPos;
let win = -1;

const colors = {};

function setup() {
  var canvas = createCanvas(cols * cellSize + 1, rows * cellSize + cellSize + 1);
  canvas.mouseClicked(doAction);


  colors.bg = color(240);
  colors.board = color(52, 92, 201);
  colors.stroke = color(0);
  colors.p1 = color(255, 224, 18);
  colors.p2 = color(224, 40, 34);
}

function draw() {
  background(colors.board); // Board color.

  playerPos = floor(mouseX / cellSize);
  stroke(colors.stroke);
  fill(colors.bg);
  rect(-1, -1, width + 2, cellSize);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const cell = board[j][i];
      switch (cell) {
        case 1:
          fill(colors.p1);
          break;
        case 2:
          fill(colors.p2);
          break;
        default:
          fill(colors.bg);
      }

      circle(
        i * cellSize + cellSize / 2,
        j * cellSize + (3 * cellSize) / 2,
        diameter
      );
    }
  }
  stroke(colors.stroke);
  for (let x = 0; x <= width; x += cellSize) {
    line(x, cellSize, x, height);
  }
  if (player == 1) {
    fill(colors.p1);
  } else if (player == 2) {
    fill(colors.p2);
  }
  circle((playerPos + 0.5) * cellSize, cellSize / 2, diameter);

  if (win != -1) {
    document.getElementById("player").removeAttribute("disabled");
    fill(0);
    if (win == 1) {
      fill(colors.p1);
    } else if (win == 2) {
      fill(colors.p2);
    }
    textAlign(CENTER, CENTER);
    textSize(64);
    if (win == 0) {
      text("It is a tie.", width / 2, cellSize / 2);
    } else {
      text(
        `${win > 1 ? "Player 2" : "Player 1"} won!`,
        width / 2,
        cellSize / 2
      );
    }
    noLoop();
  }
}


function updateBoard(newBoard) {
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const newValue = newBoard[j][i];
      board[j][i] = newValue
    }
  }
}

function resetBoard(){
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      board[j][i] = 0
    }
  }
}

function updatePlayer() {
   const selectElement = document.getElementById("player");
   const selectedValue = selectElement.value;
   player = selectedValue
   return player
}

async function playMove(action) {
  if(win > -1){
    return;
  }

  document.getElementById("player").setAttribute("disabled", "disabled");
  url = 'http://127.0.0.1:8000/play_move?action=' + action;
  const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
    };

  const response = await fetch(url, options).catch(error => {
    console.error(error); // Handle any errors
  });

  const body = await response.json();
  const play = JSON.parse(body);
  updateBoard(play.board);
  win = play.winner

}

async function newGame() {
  resetBoard();
  updatePlayer();
  document.getElementById("player").setAttribute("disabled", "disabled");
  win = -1;
  loop();
  url = 'http://127.0.0.1:8000/new_game?player=' + player;
  const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
    };

  const response = await fetch(url, options).catch(error => {
    console.error(error); // Handle any errors
  });

  const body = await response.json();
  const play = JSON.parse(body);
  updateBoard(play.board);
}


function doAction() {
    playMove(playerPos)
}
