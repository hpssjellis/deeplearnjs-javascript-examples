jQuery(document).ready(function ($) {

    // Setup navbar
    $("#nav-placeholder").load("nav.html");

    // close nav-bar dropdown when clicking outside
    $(document).click(function (event) {
        var clickover = $(event.target);
        var $navbar = $(".navbar-collapse");
        var _opened = $navbar.hasClass("in");
        if (_opened === true && !clickover.hasClass("navbar-toggle")) {
            $navbar.collapse('hide');
        }
    });


});