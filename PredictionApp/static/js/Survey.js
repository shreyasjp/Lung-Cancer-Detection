document.addEventListener("DOMContentLoaded", function () {
    const inputBoxes = document.querySelectorAll(".input-box");

    inputBoxes.forEach((inputBox) => {
        const input = inputBox.querySelector("input");
        const label = inputBox.querySelector("label");

        if(input && label){

        input.addEventListener("focus", function () {
            label.classList.add("active");
        });

        input.addEventListener("blur", function () {
            if (input.value === "") {
                label.classList.remove("active");
            }
        });
    }
    });
});